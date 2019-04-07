import time
import torch
from torch.autograd import Variable
import vqa.lib.utils as utils
import datasets.utils.metrics_utils as metrics_utils


def train(loader, model, criterion, optimizer, logger, epoch, print_freq=10):
    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')

    end = time.time()

    results = []
    bleu_score = 0
    n_sample = 0
    for i, sample in enumerate(loader):
        pred_dict = {}
        gt_dict = {}
        batch_size = sample['visual'].size(0)

        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input_visual = Variable(sample['visual'])
        input_question = Variable(sample['question'])
        target_answer = Variable(sample['answer'].cuda())

        # compute output
        output, hidden = model(input_visual, input_question)
        torch.cuda.synchronize()
        loss = criterion(output, target_answer)
        meters['loss'].update(loss.item(), n=batch_size)

        # measure accuracy
        acc1, acc2 = utils.accuracy(
            output.data, target_answer.data, topk=(1, 2))
        meters['acc1'].update(acc1.item(), n=batch_size)
        meters['acc2'].update(acc2.item(), n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            results.append({'question_id': sample['question_id'][j],
                            'answer': loader.dataset.aid_to_ans[pred[j]]})
            pred_dict[sample['question_id'][j]
                      ] = loader.dataset.aid_to_ans[pred[j]]
            gt_dict[sample['question_id'][j]
                    ] = loader.dataset.aid_to_ans[target_answer[j]]
        # bleu_batch = metrics_utils.compute_bleu_score(pred_dict, gt_dict)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #   'Bleu {bleu_batch:.4f} \t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@2 {acc2.val:.3f} ({acc2.avg:.3f})'.format(
                      epoch, i, len(loader),
                    #   bleu_batch=bleu_batch*100,
                      loss=meters['loss'], acc1=meters['acc1'], acc2=meters['acc2']))

    logger.log_meters('train', n=epoch)


def validate(loader, model, criterion, logger, epoch=0, print_freq=2):
    results = []
    bleu_score = 0
    n_sample = 0

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')

    end = time.time()
    with torch.no_grad():
        for i, sample in enumerate(loader):
            pred_dict = {}
            gt_dict = {}
            batch_size = sample['visual'].size(0)
            # input_visual   = Variable(sample['visual'].cuda(), volatile=True)
            # input_question = Variable(sample['question'].cuda(), volatile=True)
            # target_answer  = Variable(sample['answer'].cuda(), volatile=True)

            input_visual = sample['visual'].cuda()
            input_question = sample['question'].cuda()
            target_answer = sample['answer'].cuda()

            # compute output
            output, hidden = model(input_visual, input_question)
            # loss = criterion(output, target_answer)
            # meters['loss'].update(loss.item(), n=batch_size)

            # measure accuracy and record loss
            acc1, acc2 = utils.accuracy(
                output.data, target_answer.data, topk=(1, 2))
            meters['acc1'].update(acc1.item(), n=batch_size)
            meters['acc2'].update(acc2.item(), n=batch_size)

            # compute predictions for OpenEnded accuracy
            _, pred = output.data.cpu().max(1)
            target_answer = target_answer.data.cpu()
            pred.squeeze_()
            for j in range(batch_size):
                results.append({'question_id': sample['question_id'][j],
                                'answer': loader.dataset.aid_to_ans[pred[j]]})
                pred_dict[sample['question_id'][j]
                          ] = loader.dataset.aid_to_ans[pred[j]]
                try:
                    gt_dict[sample['question_id'][j]
                            ] = loader.dataset.aid_to_ans[target_answer[j]]
                except:
                    gt_dict[sample['question_id'][j]
                            ] = loader.dataset.aid_to_ans[0]

            # measure elapsed time
            meters['batch_time'].update(time.time() - end, n=batch_size)
            end = time.time()

            bleu_batch = metrics_utils.compute_bleu_score(pred_dict, gt_dict)
            if i % print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'Bleu@ {bleu_batch:.3f} \t'
                      'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                      'Acc@2 {acc2.val:.3f} ({acc2.avg:.3f})'.format(
                          i, len(loader),
                          bleu_batch=bleu_batch*100,
                          acc1=meters['acc1'], acc2=meters['acc2']))

            bleu_score += bleu_batch*batch_size
            n_sample += batch_size

    bleu_score = bleu_score / n_sample
    print(' * Bleu@ {bleu_score:.3f} Acc@1 {acc1.avg:.3f} Acc@2 {acc2.avg:.3f}'
          .format(bleu_score=bleu_score*100, acc1=meters['acc1'], acc2=meters['acc2']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, results


def test(loader, model, logger, epoch=0, print_freq=10):
    results = []
    testdev_results = []

    model.eval()
    meters = logger.reset_meters('test')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual = Variable(sample['visual'].cuda(), volatile=True)
        input_question = Variable(sample['question'].cuda(), volatile=True)

        # compute output
        output = model(input_visual, input_question)

        # compute predictions for OpenEnded accuracy
        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            item = {'question_id': sample['question_id'][j],
                    'answer': loader.dataset.aid_to_ans[pred[j]]}
            results.append(item)
            if sample['is_testdev'][j]:
                testdev_results.append(item)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                      i, len(loader), batch_time=meters['batch_time']))

    logger.log_meters('test', n=epoch)
    return results, testdev_results
